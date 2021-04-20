using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class OnlineGradientDescentTrainerSurrogate
   {
      internal class OptionsSurrogate : AveragedLinearOptionsSurrogate, ISerializationSurrogate<OnlineGradientDescentTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (OnlineGradientDescentTrainer.Options)obj;
            info.AddValue(nameof(data.LossFunction), data.LossFunction);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new OnlineGradientDescentTrainer.Options();
            SetObjectData(obj = data, info);
            info.Set(nameof(data.LossFunction), () => data.LossFunction, value => { if (value != null) data.LossFunction = value; });
            return data;
         }
      }
   }
}
