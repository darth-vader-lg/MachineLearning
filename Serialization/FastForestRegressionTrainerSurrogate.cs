using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class FastForestRegressionTrainerSurrogate
   {
      internal class OptionsSurrogate : FastForestOptionsBaseSurrogate, ISerializationSurrogate<FastForestRegressionTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (FastForestRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.ShuffleLabels), data.ShuffleLabels);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new FastForestRegressionTrainer.Options();
            obj = data;
            SetObjectData(obj, info);
            info.Set(nameof(data.ShuffleLabels), out data.ShuffleLabels);
            return data;
         }
      }
   }
}
