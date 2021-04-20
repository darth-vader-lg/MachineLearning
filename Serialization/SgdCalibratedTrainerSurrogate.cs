using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SgdCalibratedTrainerSurrogate : SgdBinaryTrainerBaseSurrogate<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<SgdCalibratedTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SgdCalibratedTrainer.Options();
            SetObjectData(data, info);
            return data;
         }
      }
   }
}
