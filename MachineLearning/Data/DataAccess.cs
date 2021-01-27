using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;

namespace MachineLearning.Data
{
   /// <summary>
   /// Vista di dati con contesto
   /// </summary>
   public class DataAccess : ChannelProvider, IDataAccess
   {
      #region Fields
      /// <summary>
      /// Contesto
      /// </summary>
      private readonly IChannelProvider context;
      /// <summary>
      /// Vista di dati
      /// </summary>
      private readonly IDataView data;
      #endregion
      #region Properties
      /// <summary>
      /// Possibilita' di shuffle
      /// </summary>
      public bool CanShuffle => data.CanShuffle;
      /// <summary>
      /// Lo schema dei dati
      /// </summary>
      public DataViewSchema Schema => data.Schema;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">Vista di dati</param>
      public DataAccess(IChannelProvider context, IDataView data)
      {
         Contracts.CheckValue(this.context = context, nameof(context));
         context.CheckValue(data, nameof(data));
         this.data = data;
      }
      /// <summary>
      /// Funzione di ottenimento del provider di canali
      /// </summary>
      /// <returns>Il provider</returns>
      protected sealed override IChannelProvider GetChannelProvider() => context;
      /// <summary>
      /// Restituisce il numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() =>
         data.GetRowCount();
      /// <summary>
      /// Restituisce un cursore
      /// </summary>
      /// <param name="columnsNeeded">Le colonne richieste</param>
      /// <param name="rand">Eventuale generatore di numeri casuali</param>
      /// <returns>Il cursore</returns>
      public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null) =>
         data.GetRowCursor(columnsNeeded, rand);
      /// <summary>
      /// Restituisce un set di cursori
      /// </summary>
      /// <param name="columnsNeeded">Le colonne richieste</param>
      /// <param name="n">Il numero desiderato di cursori</param>
      /// <param name="rand">Eventuale generatore di numeri casuali</param>
      /// <returns>I cursori</returns>
      public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
         data.GetRowCursorSet(columnsNeeded, n, rand);
      #endregion
   }
}
