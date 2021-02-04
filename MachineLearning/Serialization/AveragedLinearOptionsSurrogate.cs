using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class AveragedLinearOptionsSurrogate : OnlineLinearOptionsSurrogate
   {
      public static new void GetObjectData(object obj, SerializationInfo info)
      {
         OnlineLinearOptionsSurrogate.GetObjectData(obj, info);
         var data = (AveragedLinearOptions)obj;
         info.AddValue(nameof(data.LearningRate), data.LearningRate);
         info.AddValue(nameof(data.DecreaseLearningRate), data.DecreaseLearningRate);
         info.AddValue(nameof(data.ResetWeightsAfterXExamples), data.ResetWeightsAfterXExamples);
         info.AddValue(nameof(data.LazyUpdate), data.LazyUpdate);
         info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
         info.AddValue(nameof(data.RecencyGain), data.RecencyGain);
         info.AddValue(nameof(data.RecencyGainMultiplicative), data.RecencyGainMultiplicative);
         info.AddValue(nameof(data.Averaged), data.Averaged);
      }
      public static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (AveragedLinearOptions)OnlineLinearOptionsSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.LearningRate), out data.LearningRate);
         info.Set(nameof(data.DecreaseLearningRate), out data.DecreaseLearningRate);
         info.Set(nameof(data.ResetWeightsAfterXExamples), out data.ResetWeightsAfterXExamples);
         info.Set(nameof(data.LazyUpdate), out data.LazyUpdate);
         info.Set(nameof(data.L2Regularization), out data.L2Regularization);
         info.Set(nameof(data.RecencyGain), out data.RecencyGain);
         info.Set(nameof(data.RecencyGainMultiplicative), out data.RecencyGainMultiplicative);
         info.Set(nameof(data.Averaged), out data.Averaged);
         return data;
      }
   }
}
