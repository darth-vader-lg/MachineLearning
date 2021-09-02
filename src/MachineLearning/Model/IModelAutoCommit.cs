namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i modelli con commit automatico dei dati di training
   /// </summary>
   public interface IModelAutoCommit
   {
      #region Properties
      /// <summary>
      /// Abilitazione al commit automatico dei dati di training
      /// </summary>
      bool ModelAutoCommit { get; }
      #endregion
   }
}
