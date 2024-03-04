#pragma once
#include "Lists.h"
#include "PPO/PPOLearnerConfig.h"

namespace RLGPC {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numThreads = 8;
		int numGamesPerThread = 16;
		int minInferenceSize = 80;
		bool render = false;
		int renderDelayMS = 0;

		// Set to 0 to disable
		uint64_t timestepLimit = 0;

		int expBufferSize = 100 * 1000;
		int timestepsPerIteration = 50 * 1000;
		bool standardizeReturns = true;
		bool standardizeOBS = false; // TODO: Implement
		int maxReturnsPerStatsInc = 150;
		int stepsPerObsStatsInc = 5;

		bool autocastInference = false; // Enable torch autocast for policy inference (seems bad from my testing)
		bool halfPrecisionPolicy = false; 

		PPOLearnerConfig ppo = {};

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;

		// Set to a directory with numbered subfolders, the learner will load the subfolder with the highest number
		// If the folder is empty or does not exist, loading is skipped
		// Set empty to disable loading entirely
		std::filesystem::path checkpointLoadFolder = "checkpoints"; 

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointSaveFolder = "checkpoints"; 
		bool saveFolderAddUnixTimestamp = false; // Appends the unix time to checkpointSaveFolder

		// Save every timestep
		// Set to zero to just use timestepsPerIteration;
		int timestepsPerSave = 500 * 1000; 

		int randomSeed = 123;
		int checkpointsToKeep = 5; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		int shmBufferSize = 8 * 1024;
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "rlgymppo-cpp"; // Run name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Run name for the python metrics receiver
		std::string metricsRunName = "rlgymppo-cpp-run"; // Run name for the python metrics receiver
		
	};
}