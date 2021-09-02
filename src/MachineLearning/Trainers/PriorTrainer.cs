using Microsoft.ML;
using System;
using TTrainer = Microsoft.ML.Trainers.PriorTrainer;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Trainers.PriorModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class PriorTrainer : TrainerBase<TTransformer, TTrainer, (string LabelColumnName, string FeaturesColumnName)>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal PriorTrainer(IContextProvider<MLContext> contextProvider, (string LabelColumnName, string FeaturesColumnName) options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) =>
         context.BinaryClassification.Trainers.Prior(Options.LabelColumnName ?? "Label", Options.FeaturesColumnName ?? "Features");
      #endregion
   }
}
