using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.SdcaNonCalibratedBinaryTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.SdcaNonCalibratedBinaryTrainer;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.LinearBinaryModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaNonCalibratedBinaryTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal SdcaNonCalibratedBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.SdcaNonCalibrated(Options);
      #endregion
   }
}
