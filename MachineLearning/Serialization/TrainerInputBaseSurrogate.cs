using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class TrainerInputBaseSurrogate
   {
      public virtual void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
      {
         var data = (TrainerInputBase)obj;
         info.AddValue(nameof(data.FeatureColumnName), data.FeatureColumnName);
      }
      public virtual object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
      {
         var data = (TrainerInputBase)obj;
         info.Set(nameof(data.FeatureColumnName), out data.FeatureColumnName);
         return data;
      }
   }
}
