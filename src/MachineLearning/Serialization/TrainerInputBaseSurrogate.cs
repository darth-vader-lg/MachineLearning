using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class TrainerInputBaseSurrogate
   {
      protected static void GetObjectData(object obj, SerializationInfo info)
      {
         var data = (TrainerInputBase)obj;
         info.AddValue(nameof(data.FeatureColumnName), data.FeatureColumnName);
      }
      protected static object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (TrainerInputBase)obj;
         info.Set(nameof(data.FeatureColumnName), out data.FeatureColumnName);
         return data;
      }
   }
}
