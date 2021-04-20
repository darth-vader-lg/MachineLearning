using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class UnsupervisedTrainerInputBaseWithWeightSurrogate : TrainerInputBaseSurrogate
   {
      public static new void GetObjectData(object obj, SerializationInfo info)
      {
         TrainerInputBaseSurrogate.GetObjectData(obj, info);
         var data = (UnsupervisedTrainerInputBaseWithWeight)obj;
         info.AddValue(nameof(data.ExampleWeightColumnName), data.ExampleWeightColumnName);
      }
      public static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (UnsupervisedTrainerInputBaseWithWeight)TrainerInputBaseSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.ExampleWeightColumnName), out data.ExampleWeightColumnName);
         return data;
      }
   }
}
