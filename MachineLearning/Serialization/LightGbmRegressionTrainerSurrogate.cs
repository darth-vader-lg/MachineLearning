using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LightGbmRegressionTrainerSurrogate : LightGbmTrainerBaseSurrogate
   {
      internal class OptionsSurrogate : ISerializationSurrogate<LightGbmRegressionTrainer.Options>
      {
         private static OptionsBaseSurrogate Base => new OptionsBaseSurrogate();
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            Base.GetObjectData(obj, info, context);
            var data = (LightGbmRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.EvaluationMetric), (byte)data.EvaluationMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            obj = new LightGbmRegressionTrainer.Options();
            var data = (LightGbmRegressionTrainer.Options)Base.SetObjectData(obj, info, context, selector);
            data.EvaluationMetric = (LightGbmRegressionTrainer.Options.EvaluateMetricType)info.Set(nameof(data.EvaluationMetric), out byte _);
            return data;
         }
      }
   }
}
