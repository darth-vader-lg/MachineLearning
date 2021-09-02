using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaRegressionTrainerSurrogate : SdcaTrainerBaseSurrogate<SdcaRegressionTrainer.Options, RegressionPredictionTransformer<LinearRegressionModelParameters>, LinearRegressionModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<SdcaRegressionTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (SdcaRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.LossFunction), data.LossFunction);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaRegressionTrainer.Options();
            SetObjectData(obj = data, info);
            info.Set(nameof(data.LossFunction), () => data.LossFunction, value => { if (value != null) data.LossFunction = value; });
            return data;
         }
      }
   }
}
