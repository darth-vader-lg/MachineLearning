using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class GamTrainerBaseBaseSurrogate<TOptions, TTransformer, TPredictor>
      where TOptions : GamTrainerBase<TOptions, TTransformer, TPredictor>.OptionsBase, new()
      where TTransformer : ISingleFeaturePredictionTransformer<TPredictor>
      where TPredictor : class
   {
      internal abstract class OptionsBaseSurrogate : TrainerInputBaseWithWeightSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info)
         {
            TrainerInputBaseWithWeightSurrogate.GetObjectData(obj, info);
            var data = (GamTrainerBase<TOptions, TTransformer, TPredictor>.OptionsBase)obj;
            info.AddValue(nameof(data.EntropyCoefficient), data.EntropyCoefficient);
            info.AddValue(nameof(data.GainConfidenceLevel), data.GainConfidenceLevel);
            info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.LearningRate), data.LearningRate);
            info.AddValue(nameof(data.DiskTranspose), data.DiskTranspose);
            info.AddValue(nameof(data.MaximumBinCountPerFeature), data.MaximumBinCountPerFeature);
            info.AddValue(nameof(data.MaximumTreeOutput), data.MaximumTreeOutput);
            info.AddValue(nameof(data.GetDerivativesSampleRate), data.GetDerivativesSampleRate);
            info.AddValue(nameof(data.Seed), data.Seed);
            info.AddValue(nameof(data.MinimumExampleCountPerLeaf), data.MinimumExampleCountPerLeaf);
            info.AddValue(nameof(data.FeatureFlocks), data.FeatureFlocks);
            info.AddValue(nameof(data.EnablePruning), data.EnablePruning);
         }
         protected static new object SetObjectData(object obj, SerializationInfo info)
         {
            var data = (GamTrainerBase<TOptions, TTransformer, TPredictor>.OptionsBase)TrainerInputBaseWithWeightSurrogate.SetObjectData(obj, info);
            info.Set(nameof(data.EntropyCoefficient), out data.EntropyCoefficient);
            info.Set(nameof(data.GainConfidenceLevel), out data.GainConfidenceLevel);
            info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            info.Set(nameof(data.LearningRate), out data.LearningRate);
            info.Set(nameof(data.DiskTranspose), out data.DiskTranspose);
            info.Set(nameof(data.MaximumBinCountPerFeature), out data.MaximumBinCountPerFeature);
            info.Set(nameof(data.MaximumTreeOutput), out data.MaximumTreeOutput);
            info.Set(nameof(data.GetDerivativesSampleRate), out data.GetDerivativesSampleRate);
            info.Set(nameof(data.Seed), out data.Seed);
            info.Set(nameof(data.MinimumExampleCountPerLeaf), out data.MinimumExampleCountPerLeaf);
            info.Set(nameof(data.FeatureFlocks), out data.FeatureFlocks);
            info.Set(nameof(data.EnablePruning), out data.EnablePruning);
            return data;
         }
      }
   }
}
