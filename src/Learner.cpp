#include "Learner.h"

#include <torch/cuda.h>
#include "../libsrc/json/nlohmann/json.hpp"

RLGPC::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig _config) :
	envCreateFn(envCreateFn),
	config(_config), 
	device(at::Device(at::kCPU)) // Legally required to initialize this unfortunately
{

	if (config.timestepsPerSave == 0)
		config.timestepsPerSave = config.timestepsPerIteration;

	if (config.standardizeOBS)
		RG_ERR_CLOSE("LearnerConfig.standardizeOBS has not yet been implemented, sorry");

	RG_LOG("Learner::Learner():");
	
	if (config.saveFolderAddUnixTimestamp && !config.checkpointSaveFolder.empty())
		config.checkpointSaveFolder += "-" + std::to_string(time(0));

	RG_LOG("\tCheckpoint Load Dir: " << config.checkpointLoadFolder);
	RG_LOG("\tCheckpoint Save Dir: " << config.checkpointSaveFolder);

	torch::manual_seed(config.randomSeed);

	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(device);
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes");
	}

	{
		RG_LOG("\tCreating test environment to determine OBS size and action amount...")
		auto envCreateResult = envCreateFn();
		auto obsSet = envCreateResult.gym->Reset();
		obsSize = obsSet[0].size();
		actionAmount = envCreateResult.match->actionParser->GetActionAmount();
		RG_LOG("\t\tOBS size: " << obsSize);
		RG_LOG("\t\tAction amount: " << actionAmount);
		delete envCreateResult.gym;
		delete envCreateResult.match;
	}

	RG_LOG("\tCreating experience buffer...");
	expBuffer = new ExperienceBuffer(config.expBufferSize, config.randomSeed, device);

	RG_LOG("\tCreating PPO Learner...");
	ppo = new PPOLearner(obsSize, actionAmount, config.ppo, device);

	RG_LOG("\tCreating agent manager...");
	agentMgr = new ThreadAgentManager(ppo->policy, expBuffer, config.standardizeOBS, config.autocastInference, device);

	RG_LOG("\tCreating " << config.numThreads << " agents...");
	agentMgr->CreateAgents(envCreateFn, config.numThreads, config.numGamesPerThread);

	if (!config.checkpointLoadFolder.empty()) {
		Load();
	}
}

void RLGPC::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["cumulative_timesteps"] = totalTimesteps;
	j["cumulative_model_updates"] = ppo->cumulativeModelUpdates;
	j["epoch"] = totalEpochs;
	
	auto& rrs = j["reward_running_stats"];
	{
		rrs["mean"] = returnStats.runningMean;
		rrs["var"] = returnStats.runningVariance;
		rrs["shape"] = returnStats.shape;
		rrs["count"] = returnStats.count;
	}

	fOut << j;
}

void RLGPC::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["cumulative_timesteps"];
	ppo->cumulativeModelUpdates = j["cumulative_model_updates"];
	totalEpochs = j["epoch"];
	
	auto& rrs = j["reward_running_stats"];
	{
		returnStats = WelfordRunningStat(rrs["shape"]);
		returnStats.runningMean = rrs["mean"].get<FList>();
		returnStats.runningVariance = rrs["var"].get<FList>();
		returnStats.count = rrs["count"];
	}
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void RLGPC::Learner::Save() {
	if (config.checkpointSaveFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointSaveFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);
	RG_LOG(" > Done.");
}

void RLGPC::Learner::Load() {
	if (config.checkpointLoadFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointLoadFolder << "...");

	int64_t highest = -1;
	if (std::filesystem::is_directory(config.checkpointLoadFolder)) {
		for (auto entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
			if (entry.is_directory()) {
				auto name = entry.path().filename();
				try {
					int64_t nameVal = std::stoll(name);
					highest = RS_MAX(nameVal, highest);
				} catch (...) {}
			}
		}
	}

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointLoadFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		SaveStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder, false);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

// Prints the metrics report in a similar way to rlgym-ppo
void DisplayReport(const RLGPC::Report& report) {
	// FORMAT:
	//	blank line = print blank line
	//	'-' before name = indent with dashes and spaces
	constexpr const char* REPORT_DATA_ORDER[] = {
		"Average Episode Reward",
		"Average Step Reward",
		"Policy Entropy",
		"Value Function Loss",
		"",
		"Mean KL Divergence",
		"SB3 Clip Fraction",
		"Policy Update Magnitude",
		"Value Function Update Magnitude",
		"",
		"Collected Steps/Second",
		"Overall Steps/Second",
		"",
		"Collection Time",
		"-Policy Infer Time",
		"-Env Step Time",
		"Consumption Time",
		"-PPO Learn Time",
		// TODO: These timers don't work due to non-blocking mode
		//"--PPO Value Estimate Time",
		//"--PPO Backprop Data Time",
		//"--PPO Gradient Time",
		"Total Iteration Time",
		"",
		"Cumulative Model Updates",
		"Cumulative Timesteps",
		"",
		"Timesteps Collected"
	};

	for (const char* name : REPORT_DATA_ORDER) {
		if (strlen(name) > 0) {
			int indentLevel = 0;
			while (name[0] == '-') {
				indentLevel++;
				name++;
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}

			RG_LOG(prefix << report.SingleToString(name, true));
		} else {
			RG_LOG("");
		}
	}
}

void RLGPC::Learner::Learn() {
	RG_LOG("Learner::Learn():")
	RG_LOG("\tStarting agents...");
	agentMgr->StartAgents();

	RG_LOG("\tBeginning learning loop:");
	int64_t tsSinceSave = 0;
	Timer epochTimer = {};
	while (totalTimesteps < config.timestepLimit || config.timestepLimit == 0) {
		Report report = {};

		// Collect the desired timesteps from our agents
		RG_LOG("Collecting timesteps...");
;		GameTrajectory timesteps = agentMgr->CollectTimesteps(config.timestepsPerIteration);
		double collectionTime = epochTimer.Elapsed();
		uint64_t timestepsCollected = timesteps.size; // Use actual size instead of target size

		totalTimesteps += timestepsCollected;

		// Add it to our experience buffer, also computing GAE in the process
		RG_LOG("Adding experience...");
		AddNewExperience(timesteps);
		
		Timer ppoLearnTimer = {};
		{ // Run the actual PPO learning on the experience we have collected
			
			RG_LOG("Learning...");
			// Stop agents from inferencing if we are not on CPU
			// Otherwise PPO learn will take substantial performance hits
			bool blockAgentInference = !device.is_cpu();
			if (blockAgentInference)
				for (auto agent : agentMgr->agents)
					agent->inferenceMutex.lock();

			try {
				ppo->Learn(expBuffer, report);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Exception during PPOLearner::Learn(): " << e.what());
			}

			if (blockAgentInference)
				for (auto agent : agentMgr->agents)
					agent->inferenceMutex.unlock();

			totalEpochs += config.ppo.epochs;
		}

		double ppoLearnTime = ppoLearnTimer.Elapsed();
		double epochTime = epochTimer.Elapsed();
		epochTimer.Reset(); // Reset now otherwise we can have issues with the timer and thread input-locking
		double consumptionTime = epochTime - collectionTime;

		// Get all metrics from agent manager
		agentMgr->GetMetrics(report);

		{ // Add timers to report
			report["Total Iteration Time"] = epochTime;

			report["Collection Time"] = collectionTime;
			report["Consumption Time"] = consumptionTime;
		}

		{ // Add timestep data to report
			report["Collected Steps/Second"] = (int64_t)(timestepsCollected / collectionTime);
			report["Overall Steps/Second"] = (int64_t)(timestepsCollected / epochTime);
			report["Timesteps Collected"] = timestepsCollected;
			report["Cumulative Timesteps"] = totalTimesteps;
		}

		if (iterationCallback)
			iterationCallback(report);

		{ // Print results
			constexpr const char* DIVIDER = "======================";
			RG_LOG("\n");
			RG_LOG(DIVIDER << DIVIDER);
			RG_LOG("ITERATION COMPLETED:\n");
			DisplayReport(report);
			RG_LOG(DIVIDER << DIVIDER);
			RG_LOG("\n");
		}

		tsSinceSave += timestepsCollected;
		if (tsSinceSave > config.timestepsPerSave && !config.checkpointSaveFolder.empty()) {
			Save();
			tsSinceSave = 0;
		}

		// Reset everything
		agentMgr->ResetMetrics();
	}
	
	RG_LOG("Learner: Timestep limit of " << RG_COMMA_INT(config.timestepLimit) << " reached, stopping");
	RG_LOG("\tStopping agents...");
	agentMgr->StopAgents();
}

void RLGPC::Learner::AddNewExperience(GameTrajectory& gameTraj) {
	RG_NOGRAD;

	gameTraj.RemoveCapacity();
	auto& trajData = gameTraj.data;

	size_t count = trajData.actions.size(0);

	// Construct input to the value function estimator that includes the final state (which an action was not taken in)
	auto valInput = torch::cat({ trajData.states, torch::unsqueeze(trajData.nextStates[count - 1], 0) }).to(device, true);

	auto valPredsTensor = ppo->valueNet->Forward(valInput).cpu().flatten();
	FList valPreds = TENSOR_TO_FLIST(valPredsTensor);
	// TODO: rlgym-ppo runs torch.cuda.empty_cache() here
	
	float retStd = (config.standardizeReturns ? returnStats.GetSTD()[0] : 1);

	// Compute GAE stuff
	torch::Tensor advantages, valueTargets;
	FList returns;
	TorchFuncs::ComputeGAE(
		TENSOR_TO_FLIST(trajData.rewards),
		TENSOR_TO_FLIST(trajData.dones),
		TENSOR_TO_FLIST(trajData.truncateds),
		valPreds,
		advantages,
		valueTargets,
		returns,
		config.gaeGamma,
		config.gaeLambda,
		retStd
	);

	if (config.standardizeReturns) {
		int numToIncrement = RS_MIN(config.maxReturnsPerStatsInc, returns.size());
		returnStats.Increment(returns, numToIncrement);
	}

	auto expTensors = ExperienceTensors{
			trajData.states,
			trajData.actions,
			trajData.logProbs,
			trajData.rewards,
			trajData.nextStates,
			trajData.dones,
			trajData.truncateds,
			valueTargets,
			advantages
	};
	expBuffer->SubmitExperience(
		expTensors
	);
}