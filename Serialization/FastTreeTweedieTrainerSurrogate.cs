using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class FastTreeTweedieTrainerSurrogate
   {
      internal class OptionsSurrogate : BoostedTreeOptionsSurrogate, ISerializationSurrogate<FastTreeTweedieTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (FastTreeTweedieTrainer.Options)obj;
            info.AddValue(nameof(data.Index), data.Index);
            info.AddValue(nameof(data.EarlyStoppingMetric), data.EarlyStoppingMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new FastTreeTweedieTrainer.Options();
            obj = data;
            SetObjectData(obj, info);
            info.Set(nameof(data.Index), out data.Index);
            info.Set(nameof(data.EarlyStoppingMetric), () => data.EarlyStoppingMetric, value => data.EarlyStoppingMetric = value);
            return data;
         }
      }
   }
}
