using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.LinearMulticlassModelParameters;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.LinearMulticlassModelParameters>;
using TTrainer = Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer;
using TOptions = Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaNonCalibratedMulticlassTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal SdcaNonCalibratedMulticlassTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.SdcaNonCalibrated(Options);
      #endregion
   }
}
