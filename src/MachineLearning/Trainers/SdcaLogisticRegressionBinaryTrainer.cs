using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.SdcaLogisticRegressionBinaryTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.SdcaLogisticRegressionBinaryTrainer;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.LinearBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaLogisticRegressionBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SdcaLogisticRegressionBinaryTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal SdcaLogisticRegressionBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.SdcaLogisticRegression(Options);
      #endregion
   }
}
