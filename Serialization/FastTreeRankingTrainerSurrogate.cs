using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class FastTreeRankingTrainerSurrogate
   {
      internal class OptionsSurrogate : BoostedTreeOptionsSurrogate, ISerializationSurrogate<FastTreeRankingTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (FastTreeRankingTrainer.Options)obj;
            info.AddValue(nameof(data.CustomGains), data.CustomGains);
            info.AddValue(nameof(data.UseDcg), data.UseDcg);
            info.AddValue(nameof(data.NdcgTruncationLevel), data.NdcgTruncationLevel);
            info.AddValue(nameof(data.EarlyStoppingMetric), data.EarlyStoppingMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new FastTreeRankingTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.CustomGains), out data.CustomGains);
            info.Set(nameof(data.UseDcg), out data.UseDcg);
            info.Set(nameof(data.NdcgTruncationLevel), out data.NdcgTruncationLevel);
            info.Set(nameof(data.EarlyStoppingMetric), () => data.EarlyStoppingMetric, value => data.EarlyStoppingMetric = value);
            return data;
         }
      }
   }
}
