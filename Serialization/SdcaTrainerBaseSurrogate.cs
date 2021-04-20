using Microsoft.ML;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class SdcaTrainerBaseSurrogate<TOptions, TTransformer, TModel>
      where TOptions : SdcaTrainerBase<TOptions, TTransformer, TModel>.OptionsBase, new()
      where TTransformer : ISingleFeaturePredictionTransformer<TModel>
      where TModel : class
   {
      internal abstract class OptionsBaseSurrogate : TrainerInputBaseWithWeightSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info)
         {
            TrainerInputBaseWithWeightSurrogate.GetObjectData(obj, info);
            var data = (TOptions)obj;
            info.AddValue(nameof(data.BiasLearningRate), data.BiasLearningRate);
            info.AddValue(nameof(data.ConvergenceCheckFrequency), data.ConvergenceCheckFrequency);
            info.AddValue(nameof(data.ConvergenceTolerance), data.ConvergenceTolerance);
            info.AddValue(nameof(data.L1Regularization), data.L1Regularization);
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.MaximumNumberOfIterations), data.MaximumNumberOfIterations);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.Shuffle), data.Shuffle);
         }
         public static new object SetObjectData(object obj, SerializationInfo info)
         {
            var data = (TOptions)TrainerInputBaseWithWeightSurrogate.SetObjectData(obj, info);
            info.Set(nameof(data.BiasLearningRate), out data.BiasLearningRate);
            info.Set(nameof(data.ConvergenceCheckFrequency), out data.ConvergenceCheckFrequency);
            info.Set(nameof(data.ConvergenceTolerance), out data.ConvergenceTolerance);
            info.Set(nameof(data.L1Regularization), out data.L1Regularization);
            info.Set(nameof(data.L2Regularization), out data.L2Regularization);
            info.Set(nameof(data.MaximumNumberOfIterations), out data.MaximumNumberOfIterations);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            info.Set(nameof(data.Shuffle), out data.Shuffle);
            return data;
         }
      }
   }
}
