using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class TrainerInputBaseWithLabelSurrogate : ISerializationSurrogate<TrainerInputBaseWithLabel>
   {
      private static TrainerInputBaseSurrogate Base => new TrainerInputBaseSurrogate();
      public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
      {
         Base.GetObjectData(obj, info, context);
         var data = (TrainerInputBaseWithLabel)obj;
         info.AddValue(nameof(data.LabelColumnName), data.LabelColumnName);
      }
      public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
      {
         var data = (TrainerInputBaseWithLabel)Base.SetObjectData(obj, info, context, selector);
         info.Set(nameof(data.LabelColumnName), out data.LabelColumnName);
         return data;
      }
   }
}
