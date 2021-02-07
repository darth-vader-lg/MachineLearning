using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.LinearBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;
using TTrainer = Microsoft.ML.Trainers.SgdCalibratedTrainer;
using TOptions = Microsoft.ML.Trainers.SgdCalibratedTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SgdCalibratedTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SgdCalibratedTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal SgdCalibratedTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.SgdCalibrated(Options);
      #endregion
   }
}
