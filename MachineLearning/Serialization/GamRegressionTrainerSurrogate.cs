using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class GamRegressionTrainerSurrogate : GamTrainerBaseBaseSurrogate<GamRegressionTrainer.Options, RegressionPredictionTransformer<GamRegressionModelParameters>, GamRegressionModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<GamRegressionTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (GamRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.PruningMetrics), data.PruningMetrics);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new GamRegressionTrainer.Options();
            obj = data;
            SetObjectData(obj, info);
            info.Set(nameof(data.PruningMetrics), out data.PruningMetrics);
            return data;
         }
      }
   }
}
