using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LightGbmRankingTrainerSurrogate : LightGbmTrainerBaseSurrogate<LightGbmRankingTrainer.Options, float, RankingPredictionTransformer<LightGbmRankingModelParameters>, LightGbmRankingModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LightGbmRankingTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LightGbmRankingTrainer.Options)obj;
            info.AddValue(nameof(data.CustomGains), data.CustomGains);
            info.AddValue(nameof(data.Sigmoid), data.Sigmoid);
            info.AddValue(nameof(data.EvaluationMetric), (byte)data.EvaluationMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LightGbmRankingTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.CustomGains), out data.CustomGains);
            info.Set(nameof(data.Sigmoid), out data.Sigmoid);
            data.EvaluationMetric = (LightGbmRankingTrainer.Options.EvaluateMetricType)info.Set(nameof(data.EvaluationMetric), out byte _);
            return data;
         }
      }
   }
}
