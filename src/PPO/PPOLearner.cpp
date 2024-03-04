#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include "../Util/TorchFuncs.h"

using namespace torch;

Tensor _CopyParams(nn::Module* mod) {
	return torch::nn::utils::parameters_to_vector(mod->parameters()).cpu();
}

void _CopyPolicyParamsHalf(RLGPC::DiscretePolicy* from, RLGPC::DiscretePolicy* to) {
	auto parameters = torch::nn::utils::parameters_to_vector(from->parameters());
	torch::nn::utils::vector_to_parameters(parameters.to(torch::ScalarType::BFloat16), to->parameters());
}

RLGPC::PPOLearner::PPOLearner(int obsSpaceSize, int actSpaceSize, bool enableHalfPolicy, PPOLearnerConfig _config, Device _device)
	: config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	policy = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device);
	if (enableHalfPolicy) {
		policyHalf = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device);
		policyHalf->isHalf = true;
		_CopyPolicyParamsHalf(policy, policyHalf);
	} else {
		policyHalf = NULL;
	}

	valueNet = new ValueEstimator(obsSpaceSize, config.criticLayerSizes, device);
	policyOptimizer = new optim::Adam(policy->parameters(), optim::AdamOptions(config.policyLR));
	valueOptimizer = new optim::Adam(valueNet->parameters(), optim::AdamOptions(config.criticLR));
	valueLossFn = nn::MSELoss();
}

void RLGPC::PPOLearner::Learn(ExperienceBuffer* expBuffer, Report& report) {
	bool autocast = config.autocastLearn;
	if (autocast) RG_AUTOCAST_ON();

	int
		numIterations = 0,
		numMinibatchIterations = 0;
	float
		meanEntropy = 0,
		meanDivergence = 0,
		meanValLoss = 0;
	FList clipFractions = {};

	// Save parameters first
	auto policyBefore = _CopyParams(policy);
	auto criticBefore = _CopyParams(valueNet);

	float batchSizeRatio = config.miniBatchSize / config.batchSize;

	Timer totalTimer = {};
	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = expBuffer->GetAllBatchesShuffled(config.batchSize);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchTargetValues = batch.values;
			auto batchAdvantages = batch.advantages;

			batchActs = batchActs.view({ config.batchSize, -1 });
			policyOptimizer->zero_grad();
			valueOptimizer->zero_grad();

			for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
				Timer timer = {};

				int start = mbs;
				int stop = start + config.miniBatchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true);
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true);

				timer.Reset();
				// Compute value estimates
				auto vals = valueNet->Forward(obs);
				report.Accum("PPO Value Estimate Time", timer.Elapsed());

				timer.Reset();
				// Get policy log probs & entropy
				DiscretePolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts);
				auto logProbs = bpResult.actionLogProbs;
				auto entropy = bpResult.entropy;

				logProbs = logProbs.view_as(oldProbs);
				report.Accum("PPO Backprop Data Time", timer.Elapsed());

				// Compute PPO loss
				auto ratio = exp(logProbs - oldProbs);
				auto clipped = clamp(
					ratio, 1 - config.clipRange, 1 + config.clipRange
				);

				vals = vals.view_as(targetValues);

				// Compute policy loss
				auto policyLoss = -min(
					ratio * advantages, clipped * advantages
				).mean();
				auto valueLoss = valueLossFn(vals, targetValues);
				auto ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;

				// Compute KL divergence & clip fraction using SB3 method for reporting
				float kl;
				float clipFraction;
				{
					RG_NOGRAD;

					auto logRatio = logProbs - oldProbs;
					auto klTensor = (exp(logRatio) - 1) - logRatio;
					kl = klTensor.mean().detach().cpu().item<float>();

					clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
					clipFractions.push_back(clipFraction);
				}
				

				timer.Reset();
				// NOTE: These gradient calls are a substantial portion of learn time
				//	From my testing, they are around 61% of learn time
				//	Results will probably vary heavily depending on model size and GPU strength
				ppoLoss.backward();
				valueLoss.backward();
				report.Accum("PPO Gradient Time", timer.Elapsed());

				meanValLoss += valueLoss.cpu().detach().item<float>();
				meanDivergence += kl;
				meanEntropy += entropy.cpu().detach().item<float>();
				numMinibatchIterations += 1;
			}

			nn::utils::clip_grad_norm_(valueNet->parameters(), 0.5f);
			nn::utils::clip_grad_norm_(policy->parameters(), 0.5f);

			policyOptimizer->step();
			valueOptimizer->step();

			numIterations += 1;
		}
	}

	numIterations = RS_MAX(numIterations, 1);
	numMinibatchIterations = RS_MAX(numMinibatchIterations, 1);

	// Compute averages for the metrics that will be reported
	meanEntropy /= numMinibatchIterations;
	meanDivergence /= numMinibatchIterations;
	meanValLoss /= numMinibatchIterations;

	float meanClip = 0;
	if (!clipFractions.empty()) {
		for (float f : clipFractions)
			meanClip += f;
		meanClip /= clipFractions.size();
	}

	if (policyHalf) {
		_CopyPolicyParamsHalf(policy, policyHalf);
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = _CopyParams(policy);
	auto criticAfter = _CopyParams(valueNet);

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	float totalTime = totalTimer.Elapsed();

	// Assemble and return report
	cumulativeModelUpdates += numIterations;
	report["PPO Batch Consumption Time"] = totalTime / numIterations;
	report["Cumulative Model Updates"] = cumulativeModelUpdates;
	report["Policy Entropy"] = meanEntropy;
	report["Mean KL Divergence"] = meanDivergence;
	report["Value Function Loss"] = meanValLoss;
	report["SB3 Clip Fraction"] = meanClip;
	report["Policy Update Magnitude"] = policyUpdateMagnitude;
	report["Value Function Update Magnitude"] = criticUpdateMagnitude;
	report["PPO Learn Time"] = totalTimer.Elapsed();

	policyOptimizer->zero_grad();
	valueOptimizer->zero_grad();
	
	if (autocast) RG_AUTOCAST_OFF();
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

void TorchLoadSaveSeq(torch::nn::Sequential seq, std::filesystem::path path, c10::Device device, bool load) {
	if (load) {
		auto streamIn = std::ifstream(path, std::ios::binary);
		streamIn >> std::noskipws;

		auto sizesBefore = GetSeqSizes(seq);

		try {
			torch::load(seq, streamIn, device);
		} catch (std::exception& e) {
			RG_ERR_CLOSE(
				"Failed to load model, checkpoint may be corrupt.\n" <<
				"Exception: " << e.what()
			);
		}

		// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
		// So we need to manually check if it is the same size
		auto sizesAfter = GetSeqSizes(seq);
		if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
			std::stringstream stream;
			stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";
			
			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			RG_ERR_CLOSE(stream.str());
		}

	} else {
		auto streamOut = std::ofstream(path, std::ios::binary);
		torch::save(seq, streamOut);
	}
}

void TorchLoadSaveAll(RLGPC::PPOLearner* learner, std::filesystem::path folderPath, bool load) {

	constexpr const char* FILE_NAMES[] = {
		"PPO_POLICY.lt",
		"PPO_VALUE_NET.lt",

		"PPO_POLICY_OPTIMIZER.lt",
		"PPO_VALUE_NET_OPTIMIZER.lt",
	};

	if (load) {
		for (const char* fileName : FILE_NAMES)
			if (!std::filesystem::exists(folderPath / fileName))
				RG_ERR_CLOSE("PPOLearner: Failed to find file \"" << fileName << "\" in " << folderPath << ".")
	}

	TorchLoadSaveSeq(learner->policy->seq, folderPath / FILE_NAMES[0], learner->device, load);
	TorchLoadSaveSeq(learner->valueNet->seq, folderPath / FILE_NAMES[1], learner->device, load);

	if (load) {
		try {
			torch::serialize::InputArchive policyOptArchive;
			policyOptArchive.load_from((folderPath / FILE_NAMES[2]).string(), learner->device);
			learner->policyOptimizer->load(policyOptArchive);

			torch::serialize::InputArchive valueOptArchive;
			valueOptArchive.load_from((folderPath / FILE_NAMES[3]).string(), learner->device);
			learner->valueOptimizer->load(valueOptArchive);
		} catch (std::exception& e) {
			RG_ERR_CLOSE(
				"Failed to load optimizers, exception: " << e.what() << "\n" <<
				"Checkpoint may be corrupt."
			);
		}
	} else {
		torch::serialize::OutputArchive policyOptArchive;
		learner->policyOptimizer->save(policyOptArchive);
		policyOptArchive.save_to((folderPath / FILE_NAMES[2]).string());

		torch::serialize::OutputArchive valueOptArchive;
		learner->valueOptimizer->save(valueOptArchive);
		valueOptArchive.save_to((folderPath / FILE_NAMES[3]).string());
	}
}

void RLGPC::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	RG_LOG("PPOLearner(): Saving models to: " << folderPath);
	TorchLoadSaveAll(this, folderPath, false);
}

void RLGPC::PPOLearner::LoadFrom(std::filesystem::path folderPath, bool isFromPython)  {
	RG_LOG("PPOLearner(): Loading models from: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	if (isFromPython) {
		TorchFuncs::LoadStateDict(policy, folderPath / "PPO_POLICY.pt");
		TorchFuncs::LoadStateDict(valueNet, folderPath / "PPO_VALUE_NET.pt");

		// TODO: Load optimizer
	} else {
		TorchLoadSaveAll(this, folderPath, true);
	}

	if (policyHalf) {
		_CopyPolicyParamsHalf(policy, policyHalf);
	}

	UpdateLearningRates(config.policyLR, config.criticLR);
}

void RLGPC::PPOLearner::UpdateLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	for (auto& g : policyOptimizer->param_groups())
		static_cast<torch::optim::AdamOptions&>(g.options()).lr(policyLR);

	for (auto& g : valueOptimizer->param_groups())
		static_cast<torch::optim::AdamOptions&>(g.options()).lr(criticLR);

	std::stringstream updatedMsg;
	updatedMsg << std::scientific << "Updated learning rate to [" << policyLR << ", " << criticLR << "]";
	RG_LOG("PPOLearner: " << updatedMsg.str());
}