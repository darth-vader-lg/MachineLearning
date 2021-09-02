namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per i provider di dati
   /// </summary>
   public interface IDataStorageProvider
   {
      #region Properties
      /// <summary>
      /// Storage di dati
      /// </summary>
      IDataStorage DataStorage { get; }
      #endregion
   }
}
