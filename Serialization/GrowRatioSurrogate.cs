using Microsoft.ML.Transforms.TimeSeries;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class GrowRatioSurrogate : ISerializationSurrogate<GrowthRatio>
   {
      public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
      {
         var data = (GrowthRatio)obj;
         info.AddValue(nameof(data.Growth), data.Growth);
         info.AddValue(nameof(data.TimeSpan), data.TimeSpan);
      }

      public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
      {
         var data = new GrowthRatio();
         info.Set(nameof(data.Growth), out data.Growth);
         info.Set(nameof(data.TimeSpan), out data.TimeSpan);
         return data;
      }
   }
}
