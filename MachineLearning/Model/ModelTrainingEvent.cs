using System;

namespace MachineLearning.Model
{
   /// <summary>
   /// Evento di training del modello
   /// </summary>
   public class ModelTrainingEventArgs : EventArgs
   {
      #region Properties
      /// <summary>
      /// Task di training
      /// </summary>
      public IModelEvaluator Evaluator { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="evaluator">L'evaluator che ha generato l'evento</param>
      public ModelTrainingEventArgs(IModelEvaluator evaluator) => Evaluator = evaluator;
      #endregion
   }

   /// <summary>
   /// Delegato all'evento di training
   /// </summary>
   /// <param name="sender">Oggetto generatore dell'evento</param>
   /// <param name="e">Argomenti dell'evento</param>
   public delegate void ModelTrainingEventHandler(object sender, ModelTrainingEventArgs e);
}
