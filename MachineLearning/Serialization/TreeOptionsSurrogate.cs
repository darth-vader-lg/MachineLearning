using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class TreeOptionsSurrogate : TrainerInputBaseWithGroupIdSurrogate
   {
      protected static new void GetObjectData(object obj, SerializationInfo info)
      {
         TrainerInputBaseWithGroupIdSurrogate.GetObjectData(obj, info);
         var data = (TreeOptions)obj;
         info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
         info.AddValue(nameof(data.CompressEnsemble), data.CompressEnsemble);
         info.AddValue(nameof(data.AllowEmptyTrees), data.AllowEmptyTrees);
         info.AddValue(nameof(data.Smoothing), data.Smoothing);
         info.AddValue(nameof(data.FeatureFractionPerSplit), data.FeatureFractionPerSplit);
         info.AddValue(nameof(data.BaggingExampleFraction), data.BaggingExampleFraction);
         info.AddValue(nameof(data.BaggingSize), data.BaggingSize);
         info.AddValue(nameof(data.FeatureFraction), data.FeatureFraction);
         info.AddValue(nameof(data.NumberOfTrees), data.NumberOfTrees);
         info.AddValue(nameof(data.MinimumExampleCountPerLeaf), data.MinimumExampleCountPerLeaf);
         info.AddValue(nameof(data.NumberOfLeaves), data.NumberOfLeaves);
         info.AddValue(nameof(data.ExecutionTime), data.ExecutionTime);
         info.AddValue(nameof(data.SoftmaxTemperature), data.SoftmaxTemperature);
         info.AddValue(nameof(data.GainConfidenceLevel), data.GainConfidenceLevel);
         info.AddValue(nameof(data.FeatureReusePenalty), data.FeatureReusePenalty);
         info.AddValue(nameof(data.TestFrequency), data.TestFrequency);
         info.AddValue(nameof(data.FeatureFirstUsePenalty), data.FeatureFirstUsePenalty);
         info.AddValue(nameof(data.Bundling), data.Bundling);
         info.AddValue(nameof(data.Bias), data.Bias);
         info.AddValue(nameof(data.MinimumExamplesForCategoricalSplit), data.MinimumExamplesForCategoricalSplit);
         info.AddValue(nameof(data.MinimumExampleFractionForCategoricalSplit), data.MinimumExampleFractionForCategoricalSplit);
         info.AddValue(nameof(data.MaximumCategoricalSplitPointCount), data.MaximumCategoricalSplitPointCount);
         info.AddValue(nameof(data.MaximumCategoricalGroupCountPerNode), data.MaximumCategoricalGroupCountPerNode);
         info.AddValue(nameof(data.CategoricalSplit), data.CategoricalSplit);
         info.AddValue(nameof(data.FeatureFlocks), data.FeatureFlocks);
         info.AddValue(nameof(data.DiskTranspose), data.DiskTranspose);
         info.AddValue(nameof(data.HistogramPoolSize), data.HistogramPoolSize);
         info.AddValue(nameof(data.EntropyCoefficient), data.EntropyCoefficient);
         info.AddValue(nameof(data.FeatureSelectionSeed), data.FeatureSelectionSeed);
         info.AddValue(nameof(data.Seed), data.Seed);
         info.AddValue(nameof(data.SparsifyThreshold), data.SparsifyThreshold);
      }
      protected static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (TreeOptions)TrainerInputBaseWithGroupIdSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
         info.Set(nameof(data.CompressEnsemble), out data.CompressEnsemble);
         info.Set(nameof(data.AllowEmptyTrees), out data.AllowEmptyTrees);
         info.Set(nameof(data.Smoothing), out data.Smoothing);
         info.Set(nameof(data.FeatureFractionPerSplit), out data.FeatureFractionPerSplit);
         info.Set(nameof(data.BaggingExampleFraction), out data.BaggingExampleFraction);
         info.Set(nameof(data.BaggingSize), out data.BaggingSize);
         info.Set(nameof(data.FeatureFraction), out data.FeatureFraction);
         info.Set(nameof(data.NumberOfTrees), out data.NumberOfTrees);
         info.Set(nameof(data.MinimumExampleCountPerLeaf), out data.MinimumExampleCountPerLeaf);
         info.Set(nameof(data.NumberOfLeaves), out data.NumberOfLeaves);
         info.Set(nameof(data.ExecutionTime), out data.ExecutionTime);
         info.Set(nameof(data.SoftmaxTemperature), out data.SoftmaxTemperature);
         info.Set(nameof(data.GainConfidenceLevel), out data.GainConfidenceLevel);
         info.Set(nameof(data.FeatureReusePenalty), out data.FeatureReusePenalty);
         info.Set(nameof(data.Bundling), out data.Bundling);
         info.Set(nameof(data.Bias), out data.Bias);
         info.Set(nameof(data.MinimumExamplesForCategoricalSplit), out data.MinimumExamplesForCategoricalSplit);
         info.Set(nameof(data.MinimumExampleFractionForCategoricalSplit), out data.MinimumExampleFractionForCategoricalSplit);
         info.Set(nameof(data.MaximumCategoricalSplitPointCount), out data.MaximumCategoricalSplitPointCount);
         info.Set(nameof(data.MaximumCategoricalGroupCountPerNode), out data.MaximumCategoricalGroupCountPerNode);
         info.Set(nameof(data.CategoricalSplit), out data.CategoricalSplit);
         info.Set(nameof(data.FeatureFlocks), out data.FeatureFlocks);
         info.Set(nameof(data.DiskTranspose), out data.DiskTranspose);
         info.Set(nameof(data.HistogramPoolSize), out data.HistogramPoolSize);
         info.Set(nameof(data.EntropyCoefficient), out data.EntropyCoefficient);
         info.Set(nameof(data.FeatureSelectionSeed), out data.FeatureSelectionSeed);
         info.Set(nameof(data.Seed), out data.Seed);
         info.Set(nameof(data.SparsifyThreshold), out data.SparsifyThreshold);
         return data;
      }
   }
}
