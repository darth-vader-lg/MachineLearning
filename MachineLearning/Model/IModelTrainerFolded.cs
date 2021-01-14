namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i trainer di modello con folding
   /// </summary>
   public interface IModelTrainerFolded : IModelTrainer
   {
      #region Properties
      /// <summary>
      /// Numero di folds di training
      /// </summary>
      int NumFolds { get; }
      #endregion
   }
}
