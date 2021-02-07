using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaNonCalibratedBinaryTrainerSurrogate : SdcaBinaryTrainerBaseSurrogate<LinearBinaryModelParameters>
   {
      internal class OptionsSurrogate : BinaryOptionsBaseSurrogate, ISerializationSurrogate<SdcaNonCalibratedBinaryTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (SdcaNonCalibratedBinaryTrainer.Options)obj;
            info.AddValue(nameof(data.LossFunction), data.LossFunction);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaNonCalibratedBinaryTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.LossFunction), () => data.LossFunction, value => { if (value != null) data.LossFunction = value; });
            return data;
         }
      }
   }
}
