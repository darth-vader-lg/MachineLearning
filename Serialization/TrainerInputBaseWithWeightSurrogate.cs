using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class TrainerInputBaseWithWeightSurrogate : TrainerInputBaseWithLabelSurrogate
   {
      public static new void GetObjectData(object obj, SerializationInfo info)
      {
         TrainerInputBaseWithLabelSurrogate.GetObjectData(obj, info);
         var data = (TrainerInputBaseWithWeight)obj;
         info.AddValue(nameof(data.ExampleWeightColumnName), data.ExampleWeightColumnName);
      }
      public static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (TrainerInputBaseWithWeight)TrainerInputBaseWithLabelSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.ExampleWeightColumnName), out data.ExampleWeightColumnName);
         return data;
      }
   }
}
