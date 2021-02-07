using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LbfgsPoissonRegressionTrainerSurrogate : LbfgsTrainerBaseSurrogate<LbfgsPoissonRegressionTrainer.Options, RegressionPredictionTransformer<PoissonRegressionModelParameters>, PoissonRegressionModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LbfgsPoissonRegressionTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LbfgsPoissonRegressionTrainer.Options();
            SetObjectData(obj, info);
            return data;
         }
      }
   }
}
