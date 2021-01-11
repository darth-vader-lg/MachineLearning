using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LightGbmRegressionTrainerSurrogate : LightGbmTrainerBaseSurrogate<LightGbmRegressionTrainer.Options, float, RegressionPredictionTransformer<LightGbmRegressionModelParameters>, LightGbmRegressionModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LightGbmRegressionTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LightGbmRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.EvaluationMetric), (byte)data.EvaluationMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LightGbmRegressionTrainer.Options();
            obj = data;
            SetObjectData(obj, info);
            data.EvaluationMetric = (LightGbmRegressionTrainer.Options.EvaluateMetricType)info.Set(nameof(data.EvaluationMetric), out byte _);
            return data;
         }
      }
   }
}
