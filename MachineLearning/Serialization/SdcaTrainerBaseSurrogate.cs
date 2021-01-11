using Microsoft.ML;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaTrainerBaseSurrogate<TOptions, TTransformer, TModel>
      where TOptions : SdcaTrainerBase<TOptions, TTransformer, TModel>.OptionsBase, new()
      where TTransformer : ISingleFeaturePredictionTransformer<TModel>
      where TModel : class
   {
      internal class OptionsBaseSurrogate
      {
         private static TrainerInputBaseWithWeightSurrogate Base => new TrainerInputBaseWithWeightSurrogate();
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            Base.GetObjectData(obj, info, context);
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
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = (TOptions)Base.SetObjectData(obj, info, context, selector);
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
