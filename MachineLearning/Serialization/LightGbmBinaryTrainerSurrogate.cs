using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LightGbmBinaryTrainerSurrogate : LightGbmTrainerBaseSurrogate<LightGbmBinaryTrainer.Options, float, BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>, CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LightGbmBinaryTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LightGbmBinaryTrainer.Options)obj;
            info.AddValue(nameof(data.UnbalancedSets), data.UnbalancedSets);
            info.AddValue(nameof(data.WeightOfPositiveExamples), data.WeightOfPositiveExamples);
            info.AddValue(nameof(data.Sigmoid), data.Sigmoid);
            info.AddValue(nameof(data.EvaluationMetric), (byte)data.EvaluationMetric);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LightGbmBinaryTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.UnbalancedSets), out data.UnbalancedSets);
            info.Set(nameof(data.WeightOfPositiveExamples), out data.WeightOfPositiveExamples);
            info.Set(nameof(data.Sigmoid), out data.Sigmoid);
            data.EvaluationMetric = (LightGbmBinaryTrainer.Options.EvaluateMetricType)info.Set(nameof(data.EvaluationMetric), out byte _);
            return data;
         }
      }
   }
}
