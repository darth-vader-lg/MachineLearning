namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i providers di opzioni di testo
   /// </summary>
   public interface IDataTextOptionsProvider
   {
      #region Properties
      /// <summary>
      /// Opzioni di caricamento testi
      /// </summary>
      TextLoaderOptions TextOptions { get; }
      #endregion
   }
}
