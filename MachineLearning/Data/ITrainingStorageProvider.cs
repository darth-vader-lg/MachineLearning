namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per i provider di dati di training
   /// </summary>
   public interface ITrainingStorageProvider
   {
      #region Properties
      /// <summary>
      /// Storage di dati di training
      /// </summary>
      IDataStorage TrainingStorage { get; }
      #endregion
   }
}
