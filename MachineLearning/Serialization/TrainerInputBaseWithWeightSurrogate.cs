using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class TrainerInputBaseWithWeightSurrogate : ISerializationSurrogate<TrainerInputBaseWithWeight>
   {
      private static TrainerInputBaseWithLabelSurrogate Base => new TrainerInputBaseWithLabelSurrogate();
      public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
      {
         Base.GetObjectData(obj, info, context);
         var data = (TrainerInputBaseWithWeight)obj;
         info.AddValue(nameof(data.ExampleWeightColumnName), data.ExampleWeightColumnName);
      }
      public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
      {
         var data = (TrainerInputBaseWithWeight)Base.SetObjectData(obj, info, context, selector);
         info.Set(nameof(data.ExampleWeightColumnName), out data.ExampleWeightColumnName);
         return data;
      }
   }
}
