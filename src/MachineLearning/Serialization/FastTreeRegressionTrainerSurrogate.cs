using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class FastTreeRegressionTrainerSurrogate
   {
      internal class OptionsSurrogate : BoostedTreeOptionsSurrogate, ISerializationSurrogate<FastTreeRegressionTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (FastTreeRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.EarlyStoppingMetric), data.EarlyStoppingMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new FastTreeRegressionTrainer.Options();
            obj = data;
            SetObjectData(obj, info);
            info.Set(nameof(data.EarlyStoppingMetric), () => data.EarlyStoppingMetric, value => data.EarlyStoppingMetric = value);
            return data;
         }
      }
   }
}
