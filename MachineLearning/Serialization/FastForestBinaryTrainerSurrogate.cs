using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class FastForestBinaryTrainerSurrogate
   {
      internal class OptionsSurrogate : FastForestOptionsBaseSurrogate, ISerializationSurrogate<FastForestBinaryTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (FastForestBinaryTrainer.Options)obj;
            info.AddValue(nameof(data.MaximumOutputMagnitudePerTree), data.MaximumOutputMagnitudePerTree);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new FastForestBinaryTrainer.Options();
            obj = data;
            SetObjectData(obj, info);
            info.Set(nameof(data.MaximumOutputMagnitudePerTree), out data.MaximumOutputMagnitudePerTree);
            return data;
         }
      }
   }
}
