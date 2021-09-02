using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class BoostedTreeOptionsSurrogate : TreeOptionsSurrogate
   {
      protected static new void GetObjectData(object obj, SerializationInfo info)
      {
         TreeOptionsSurrogate.GetObjectData(obj, info);
         var data = (BoostedTreeOptions)obj;
         info.AddValue(nameof(data.BestStepRankingRegressionTrees), data.BestStepRankingRegressionTrees);
         info.AddValue(nameof(data.FilterZeroLambdas), data.FilterZeroLambdas);
         info.AddValue(nameof(data.RandomStart), data.RandomStart);
         info.AddValue(nameof(data.MaximumTreeOutput), data.MaximumTreeOutput);
         info.AddValue(nameof(data.WriteLastEnsemble), data.WriteLastEnsemble);
         info.AddValue(nameof(data.GetDerivativesSampleRate), data.GetDerivativesSampleRate);
         info.AddValue(nameof(data.DropoutRate), data.DropoutRate);
         info.AddValue(nameof(data.Shrinkage), data.Shrinkage);
         info.AddValue(nameof(data.LearningRate), data.LearningRate);
         info.AddValue(nameof(data.PruningThreshold), data.PruningThreshold);
         info.AddValue(nameof(data.UseTolerantPruning), data.UseTolerantPruning);
         info.AddValue(nameof(data.EnablePruning), data.EnablePruning);
         info.AddValue(nameof(data.OptimizationAlgorithm), data.OptimizationAlgorithm);
         info.AddValue(nameof(data.MinimumStepSize), data.MinimumStepSize);
         info.AddValue(nameof(data.MaximumNumberOfLineSearchSteps), data.MaximumNumberOfLineSearchSteps);
         info.AddValue(nameof(data.UseLineSearch), data.UseLineSearch);
         info.AddValue(nameof(data.PruningWindowSize), data.PruningWindowSize);
         info.AddValue(nameof(data.EarlyStoppingRule), data.EarlyStoppingRule);
      }
      protected static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (BoostedTreeOptions)TreeOptionsSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.BestStepRankingRegressionTrees), out data.BestStepRankingRegressionTrees);
         info.Set(nameof(data.FilterZeroLambdas), out data.FilterZeroLambdas);
         info.Set(nameof(data.RandomStart), out data.RandomStart);
         info.Set(nameof(data.MaximumTreeOutput), out data.MaximumTreeOutput);
         info.Set(nameof(data.WriteLastEnsemble), out data.WriteLastEnsemble);
         info.Set(nameof(data.GetDerivativesSampleRate), out data.GetDerivativesSampleRate);
         info.Set(nameof(data.DropoutRate), out data.DropoutRate);
         info.Set(nameof(data.Shrinkage), out data.Shrinkage);
         info.Set(nameof(data.LearningRate), out data.LearningRate);
         info.Set(nameof(data.PruningThreshold), out data.PruningThreshold);
         info.Set(nameof(data.UseTolerantPruning), out data.UseTolerantPruning);
         info.Set(nameof(data.EnablePruning), out data.EnablePruning);
         info.Set(nameof(data.OptimizationAlgorithm), out data.OptimizationAlgorithm);
         info.Set(nameof(data.MinimumStepSize), out data.MinimumStepSize);
         info.Set(nameof(data.MaximumNumberOfLineSearchSteps), out data.MaximumNumberOfLineSearchSteps);
         info.Set(nameof(data.UseLineSearch), out data.UseLineSearch);
         info.Set(nameof(data.PruningWindowSize), out data.PruningWindowSize);
         info.Set(nameof(data.EarlyStoppingRule), () => data.EarlyStoppingRule, value => data.EarlyStoppingRule = value);
         return data;
      }
   }
}
