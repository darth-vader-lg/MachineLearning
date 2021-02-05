using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class LightGbmMulticlassTrainerSurrogate : LightGbmTrainerBaseSurrogate<LightGbmMulticlassTrainer.Options, VBuffer<float>, MulticlassPredictionTransformer<OneVersusAllModelParameters>, OneVersusAllModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LightGbmMulticlassTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LightGbmMulticlassTrainer.Options)obj;
            info.AddValue(nameof(data.UnbalancedSets), data.UnbalancedSets);
            info.AddValue(nameof(data.UseSoftmax), data.UseSoftmax);
            info.AddValue(nameof(data.Sigmoid), data.Sigmoid);
            info.AddValue(nameof(data.EvaluationMetric), data.EvaluationMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LightGbmMulticlassTrainer.Options();
            SetObjectData(obj = data, info);
            info.Set(nameof(data.UnbalancedSets), out data.UnbalancedSets);
            info.Set(nameof(data.UseSoftmax), out data.UseSoftmax);
            info.Set(nameof(data.Sigmoid), out data.Sigmoid);
            info.Set(nameof(data.EvaluationMetric), out data.EvaluationMetric);
            return data;
         }
      }
   }
}
