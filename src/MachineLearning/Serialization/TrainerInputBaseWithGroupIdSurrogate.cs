using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class TrainerInputBaseWithGroupIdSurrogate : TrainerInputBaseWithWeightSurrogate
   {
      protected static new void GetObjectData(object obj, SerializationInfo info)
      {
         TrainerInputBaseWithWeightSurrogate.GetObjectData(obj, info);
         var data = (TrainerInputBaseWithGroupId)obj;
         info.AddValue(nameof(data.RowGroupColumnName), data.RowGroupColumnName);
      }
      protected static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (TrainerInputBaseWithGroupId)TrainerInputBaseWithWeightSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.RowGroupColumnName), out data.RowGroupColumnName);
         return data;
      }
   }
}
