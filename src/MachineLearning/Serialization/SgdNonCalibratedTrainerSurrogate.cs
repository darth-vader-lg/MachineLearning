using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SgdNonCalibratedTrainerSurrogate : SgdBinaryTrainerBaseSurrogate<LinearBinaryModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<SgdNonCalibratedTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (SgdNonCalibratedTrainer.Options)obj;
            info.AddValue(nameof(data.LossFunction), data.LossFunction);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SgdNonCalibratedTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.LossFunction), () => data.LossFunction, value => { if (value != null) data.LossFunction = value; });
            return data;
         }
      }
   }
}
