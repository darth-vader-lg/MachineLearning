using Microsoft.ML;
using System;
using TTrainer = Microsoft.ML.Trainers.NaiveBayesMulticlassTrainer;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.NaiveBayesMulticlassModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class NaiveBayesMulticlassTrainer : TrainerBase<TTransformer, TTrainer, (string LabelColumnName, string FeaturesColumnName)>
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
      protected override TTrainer CreateTrainer(MLContext context) =>
         context.MulticlassClassification.Trainers.NaiveBayes(Options.LabelColumnName ?? "Label", Options.FeaturesColumnName ?? "Features");
      #endregion
   }
}
