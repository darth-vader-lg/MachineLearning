using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class OnlineLinearOptionsSurrogate : TrainerInputBaseWithLabelSurrogate
   {
      public static new void GetObjectData(object obj, SerializationInfo info)
      {
         TrainerInputBaseWithLabelSurrogate.GetObjectData(obj, info);
         var data = (OnlineLinearOptions)obj;
         info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
         info.AddValue(nameof(data.InitialWeightsDiameter), data.InitialWeightsDiameter);
         info.AddValue(nameof(data.Shuffle), data.Shuffle);
      }
      public static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (OnlineLinearOptions)TrainerInputBaseWithLabelSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
         info.Set(nameof(data.InitialWeightsDiameter), out data.InitialWeightsDiameter);
         info.Set(nameof(data.Shuffle), out data.Shuffle);
         return data;
      }
   }
}
