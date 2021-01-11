using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class TrainerInputBaseWithGroupIdSurrogate : ISerializationSurrogate<TrainerInputBaseWithGroupId>
   {
      private static TrainerInputBaseWithWeightSurrogate Base => new TrainerInputBaseWithWeightSurrogate();
      public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
      {
         Base.GetObjectData(obj, info, context);
         var data = (TrainerInputBaseWithGroupId)obj;
         info.AddValue(nameof(data.RowGroupColumnName), data.RowGroupColumnName);
      }
      public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
      {
         var data = (TrainerInputBaseWithGroupId)Base.SetObjectData(obj, info, context, selector);
         info.Set(nameof(data.RowGroupColumnName), out data.RowGroupColumnName);
         return data;
      }
   }
}
