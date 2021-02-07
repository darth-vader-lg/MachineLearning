using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class SgdBinaryTrainerBaseSurrogate<TModel> where TModel : class
   {
      internal abstract class OptionsBaseSurrogate : TrainerInputBaseWithWeightSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info)
         {
            TrainerInputBaseWithWeightSurrogate.GetObjectData(obj, info);
            var data = (SgdBinaryTrainerBase<TModel>.OptionsBase)obj;
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.ConvergenceTolerance), data.ConvergenceTolerance);
            info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
            info.AddValue(nameof(data.LearningRate), data.LearningRate);
            info.AddValue(nameof(data.Shuffle), data.Shuffle);
            info.AddValue(nameof(data.PositiveInstanceWeight), data.PositiveInstanceWeight);
            info.AddValue(nameof(data.CheckFrequency), data.CheckFrequency);
         }
         public static new object SetObjectData(object obj, SerializationInfo info)
         {
            var data = (SgdBinaryTrainerBase<TModel>.OptionsBase)TrainerInputBaseWithWeightSurrogate.SetObjectData(obj, info);
            info.Set(nameof(data.L2Regularization), out data.L2Regularization);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            info.Set(nameof(data.ConvergenceTolerance), out data.ConvergenceTolerance);
            info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
            info.Set(nameof(data.LearningRate), out data.LearningRate);
            info.Set(nameof(data.Shuffle), out data.Shuffle);
            info.Set(nameof(data.PositiveInstanceWeight), out data.PositiveInstanceWeight);
            info.Set(nameof(data.CheckFrequency), out data.CheckFrequency);
            return data;
         }
      }
   }
}
