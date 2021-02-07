using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class GamBinaryTrainerSurrogate : GamTrainerBaseBaseSurrogate<GamBinaryTrainer.Options, BinaryPredictionTransformer<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>, CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<GamBinaryTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (GamBinaryTrainer.Options)obj;
            info.AddValue(nameof(data.UnbalancedSets), data.UnbalancedSets);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new GamBinaryTrainer.Options();
            SetObjectData(obj, info);
            info.Set(nameof(data.UnbalancedSets), out data.UnbalancedSets);
            return data;
         }
      }
   }
}
