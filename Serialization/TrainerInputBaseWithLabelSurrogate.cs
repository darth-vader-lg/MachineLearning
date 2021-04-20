using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class TrainerInputBaseWithLabelSurrogate : TrainerInputBaseSurrogate
   {
      protected static new void GetObjectData(object obj, SerializationInfo info)
      {
         TrainerInputBaseSurrogate.GetObjectData(obj, info);
         var data = (TrainerInputBaseWithLabel)obj;
         info.AddValue(nameof(data.LabelColumnName), data.LabelColumnName);
      }
      protected static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (TrainerInputBaseWithLabel)TrainerInputBaseSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.LabelColumnName), out data.LabelColumnName);
         return data;
      }
   }
}
