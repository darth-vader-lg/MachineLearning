using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LbfgsLogisticRegressionBinaryTrainerSurrogate : LbfgsTrainerBaseSurrogate<LbfgsLogisticRegressionBinaryTrainer.Options, BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>, CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LbfgsLogisticRegressionBinaryTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LbfgsLogisticRegressionBinaryTrainer.Options)obj;
            info.AddValue(nameof(data.ShowTrainingStatistics), data.ShowTrainingStatistics);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LbfgsLogisticRegressionBinaryTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.ShowTrainingStatistics), out data.ShowTrainingStatistics);
            return data;
         }
      }
   }
}
