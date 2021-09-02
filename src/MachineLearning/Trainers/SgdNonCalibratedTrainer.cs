using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.SgdNonCalibratedTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.SgdNonCalibratedTrainer;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Trainers.LinearBinaryModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SgdNonCalibratedTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SgdNonCalibratedTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal SgdNonCalibratedTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.SgdNonCalibrated(Options);
      #endregion
   }
}
