using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.NaiveBayesMulticlassModelParameters;
using MSTrainer = Microsoft.ML.Trainers.NaiveBayesMulticlassTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class NaiveBayesMulticlassTrainer :
      TrainerBase<MSModel, MSTrainer, (string LabelColumnName, string FeaturesColumnName)>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal NaiveBayesMulticlassTrainer(IContextProvider<MLContext> contextProvider, (string LabelColumnName, string FeaturesColumnName) options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) =>
         context.MulticlassClassification.Trainers.NaiveBayes(Options.LabelColumnName ?? "Label", Options.FeaturesColumnName ?? "Features");
      #endregion
   }
}
