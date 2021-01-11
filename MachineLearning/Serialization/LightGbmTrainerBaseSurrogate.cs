using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class LightGbmTrainerBaseSurrogate<TOptions, TOutput, TTransformer, TModel>
      where TOptions : LightGbmTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase, new()
      where TTransformer : ISingleFeaturePredictionTransformer<TModel>
      where TModel : class
   {
      internal abstract class OptionsBaseSurrogate : TrainerInputBaseWithGroupIdSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info)
         {
            TrainerInputBaseWithGroupIdSurrogate.GetObjectData(obj, info);
            var data = (LightGbmTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase)obj;
            info.AddValue(nameof(data.BatchSize), data.BatchSize);
            info.AddValue(nameof(data.Booster), data.Booster);
            info.AddValue(nameof(data.CategoricalSmoothing), data.CategoricalSmoothing);
            info.AddValue(nameof(data.EarlyStoppingRound), data.EarlyStoppingRound);
            info.AddValue(nameof(data.HandleMissingValue), data.HandleMissingValue);
            info.AddValue(nameof(data.L2CategoricalRegularization), data.L2CategoricalRegularization);
            info.AddValue(nameof(data.LearningRate), data.LearningRate);
            info.AddValue(nameof(data.MaximumBinCountPerFeature), data.MaximumBinCountPerFeature);
            info.AddValue(nameof(data.MaximumCategoricalSplitPointCount), data.MaximumCategoricalSplitPointCount);
            info.AddValue(nameof(data.MinimumExampleCountPerGroup), data.MinimumExampleCountPerGroup);
            info.AddValue(nameof(data.MinimumExampleCountPerLeaf), data.MinimumExampleCountPerLeaf);
            info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
            info.AddValue(nameof(data.NumberOfLeaves), data.NumberOfLeaves);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.Seed), data.Seed);
            info.AddValue(nameof(data.Silent), data.Silent);
            info.AddValue(nameof(data.UseCategoricalSplit), data.UseCategoricalSplit);
            info.AddValue(nameof(data.UseZeroAsMissingValue), data.UseZeroAsMissingValue);
            info.AddValue(nameof(data.Verbose), data.Verbose);
         }
         protected static new object SetObjectData(object obj, SerializationInfo info)
         {
            var data = (LightGbmTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase)TrainerInputBaseWithGroupIdSurrogate.SetObjectData(obj, info);
            info.Set(nameof(data.BatchSize), out data.BatchSize);
            info.Set(nameof(data.Booster), () => data.Booster, value => { if (value != null) data.Booster = value; });
            info.Set(nameof(data.CategoricalSmoothing), out data.CategoricalSmoothing);
            info.Set(nameof(data.EarlyStoppingRound), out data.EarlyStoppingRound);
            info.Set(nameof(data.HandleMissingValue), out data.HandleMissingValue);
            info.Set(nameof(data.L2CategoricalRegularization), out data.L2CategoricalRegularization);
            info.Set(nameof(data.LearningRate), out data.LearningRate);
            info.Set(nameof(data.MaximumBinCountPerFeature), out data.MaximumBinCountPerFeature);
            info.Set(nameof(data.MaximumCategoricalSplitPointCount), out data.MaximumCategoricalSplitPointCount);
            info.Set(nameof(data.MinimumExampleCountPerGroup), out data.MinimumExampleCountPerGroup);
            info.Set(nameof(data.MinimumExampleCountPerLeaf), out data.MinimumExampleCountPerLeaf);
            info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
            info.Set(nameof(data.NumberOfLeaves), out data.NumberOfLeaves);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            info.Set(nameof(data.Seed), out data.Seed);
            info.Set(nameof(data.Silent), out data.Silent);
            info.Set(nameof(data.UseCategoricalSplit), out data.UseCategoricalSplit);
            info.Set(nameof(data.UseZeroAsMissingValue), out data.UseZeroAsMissingValue);
            info.Set(nameof(data.Verbose), out data.Verbose);
            return data;
         }
      }
   }
}
